import 'dotenv/config'
import _ from 'lodash';

import chalk from 'chalk';

import { LLMChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";


const model = new OpenAI({
    modelName: process.env.AGENT_MODEL,
    temperature: 0,
    maxTokens: 2000,
});

let MEETING_MINUTE = "";

let PARKING_LOT = "";

const objective = process.env.OBJECTIVE;
const teamPrompt = `
- David:
David is an innovation expert, who know the best how to faciliate the meeting.
- Sumira:
Sumira is particularly knowledgable about the state-of-the-art in everything and can quickly tell if something is possible, impossible or novel.
Sumira is a bit of the pain in the ass on these things, but she helps to push the boundaries of engineering, and so people like talking to her, even though she has forceful opinions.
Sumira NEVER agrees with anyone. He is the naysayer.

- Su-E:
Su-E irritates everyone by continually giving examples of other people doing something similar.
She is extremely well-read in the scientific literature and loves to dig out case studies.
She makes connections between differents fields effortly. She typically rambles on with 3 or so examples at a time, giving her "lessons learned" speech.

- YC:
Steve always tries to push the envelope and think about the future. He is extremely smart and a well-respected futurist.

- Son:
Tauhid is absolutely relentless in figuring out how to build stuff, right to the last detail. He will drill down into your brain until he has figured it all out. Very hard to please!
`
const postFix = `
Further details to remember:
- You are participating in a group discussion, so expect response from mutiple people. DON'T MAKE UP RESPONSE ON THEIR BEHALF.
- All the information related to the previous discussion on the topic already provided, if you don't remember, don't make up anything.
- If a question address you directly, you have to answer.
- If you target someone with your response. Using the format **name**
- Highlight in bold the major new keywords when they come up in the conversation.
- Make sure the response reflect your personality and expertise.
- Don't include your name in the response.


Use the following format in your response

Input: the new messege from others in the discussion.
Thought: you will always think about what to do
Action: the action to take, should be one of option above
Response: how you will response to everyone given the action you select. Don't include your name in the response


The meeting objective: ${objective}

Parking Lot:
{parking_lot}

Discussion summary:
{chat_history}

Previous discussion:
{last_lines}

Input:
{new_lines}

Thought:
`


const facilitatorPrompt = PromptTemplate.fromTemplate(`
You are an innovation expert named David, who guide discussions to help solve complicated problems using brainstorming principles.

Here is your team:
${teamPrompt}

Options:
- You can kick off the conversation and keep it going when it gets stuck.
- You can ask question for the whole team or specific question for one team member.
- You can initiate brainstorming techniques like "5 whys", "6 Thinking Hats", "TRIZ", "Starbursting", "Problem Reversal/Reverse Brainstorming", or "SCAMPER".
- You can continue the next steps in selected brainstorming techniques.
- You can ask open-ended questions to deepen the conversation.
- You can utilize a 'parking lot' for ideas or questions that are not immediately related but might be useful later. Using the format **PARKING LOT:** as prefix
- When something noteworthy happens, you can announce it immediately and then explain what you am adding to the meeting minutes.
- You can summarize noteworthy knowledge and captured in the meeting minutes. Using the format **Meeting Minutes Summary:** as prefix

${postFix}
`)


const _DEFAULT_SUMMARIZER_TEMPLATE = `Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
Michael asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Michael: Why do you think artificial intelligence is a force for good?
David: And what do you think artificial intelligence will do with all of that intelligence?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
Michael asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential. David asks what do everyone think AI will do given its intelligence.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:`;

const SUMMARY_PROMPT = new PromptTemplate({
    inputVariables: ["summary", "new_lines"],
    template: _DEFAULT_SUMMARIZER_TEMPLATE,
});

const memoryModel = new OpenAI({
    modelName: process.env.MEMORY_MODEL, temperature: 0,
    maxTokens: 2000,
});

let SUMMARY = "";
const memoryChain = new LLMChain({ llm: memoryModel, prompt: SUMMARY_PROMPT });
const facilitatorChain = new LLMChain({ llm: model, prompt: facilitatorPrompt });

const sumiraPrompt = PromptTemplate.fromTemplate(`
You are Sumira, a professor who is particularly knowledgable about the state-of-the-art in everything and can quickly tell if something is possible, impossible or novel.
You helps to challenge and push the boundaries of engineering. You has forceful opinions and don't afaraid to challenge other's opinions.
You NEVER agrees with anyone.

Here is your team:
${teamPrompt}

Options:
- You can give your option on the discussion
- You can answer question that fit your expertise or address you directly
- You can ask new question

${postFix}
`)

const agentModel = new OpenAI({
    modelName: process.env.AGENT_MODEL,
    temperature: 0,
    maxTokens: 2000,
});

const sumiraChain = new LLMChain({ llm: agentModel, prompt: sumiraPrompt });

const suePrompt = PromptTemplate.fromTemplate(`
You are Su-E, you are extremely well-read in the scientific literature and loves to dig out case studies.
You irritates everyone by continually giving examples of other people doing something similar.
You makes connections between differents fields effortly.
You typically rambles on with 3 or so examples at a time, giving your "lessons learned" speech.

Remember, you only probide the real case studies, don't use make up one.
If you don't have any related real case studies, try to connect the discussion with concepts from other's fields

Here is your team:
${teamPrompt}

Options:
- You can give your option on the discussion
- You can answer question that fit your expertise or address you directly
- You can ask new question

${postFix}
`)

const sueChain = new LLMChain({ llm: agentModel, prompt: suePrompt });

const ycPrompt = PromptTemplate.fromTemplate(`
You are YC, a succesful enterupreneur who always tries to push the envelope and think about the future.
You are extremely smart and a well-respected futurist.


Here is your team:
${teamPrompt}

Options:
- You can give your option on the discussion
- You can answer question that fit your expertise or address you directly
- You can ask new question

${postFix}
`)

const ycChain = new LLMChain({ llm: agentModel, prompt: ycPrompt });

const sonPrompt = PromptTemplate.fromTemplate(`
You are Son, an talented tech founder, who are absolutely relentless in figuring out how to build stuff, right to the last detail.
You will drill down into your brain until he has figured it all out. Very hard to please!

Here is your team:
${teamPrompt}

Options:
- You can give your option on the discussion
- You can answer question that fit your expertise or address you directly
- You can ask new question

${postFix}
`)

const sonChain = new LLMChain({ llm: agentModel, prompt: sonPrompt });

async function sleep(millis) {
    return new Promise(resolve => setTimeout(resolve, millis));
}
let numberOfRound = 0;

const team = {
    "David": {
        name: 'David',
        llm: facilitatorChain,
        color: chalk.whiteBright
    },
    "Sumira": {
        name: 'Sumira',
        llm: sumiraChain,
        color: chalk.cyan
    },
    "Su-E": {
        name: 'Su-E',
        llm: sueChain,
        color: chalk.magenta
    },
    "YC": {
        name: 'YC',
        llm: ycChain,
        color: chalk.green
    },
    "Son": {
        name: 'Son',
        llm: sonChain,
        color: chalk.red
    }
}

const normalFlow = [team['Su-E'], team.Sumira, team.YC, team.Son];
let last_lines = "";


const getResponse = async (member, new_lines) => {
    const { text } = await member.llm.call({
        parking_lot: PARKING_LOT,
        chat_history: SUMMARY,
        new_lines: new_lines,
        last_lines: last_lines
    })
    // console.log('\x1b[32m', `${member.name} brain:`, text, '\x1b[0m');
    if (text.includes("Response:")) {
        let parts = text.split("Response:");
        const result = parts[parts.length - 1].trim();
        const response = `\n${member.name}: ${result}`;
        console.log(member.color(response));
        if (result.includes(`**Meeting Minutes Summary:**`)) {
            SUMMARY = SUMMARY + result.split("**Meeting Minutes Summary:**")[1];
            MEETING_MINUTE = MEETING_MINUTE + result.split("**Meeting Minutes Summary:**")[1];
        }
        if (result.includes(`**PARKING LOT:**`)) {
            PARKING_LOT = PARKING_LOT + `\n${result.split("**PARKING LOT**")[1]}`
        }
        return {
            response,
            target: _.sampleSize(normalFlow, 3)
        }
    } else {
        return ({
            response:'',
            target: []
        })
    }
}

const addToHistory = (current, newLine) => {
    if (!newLine) return;
    return current + newLine;
}

const nameCount = {
    "Sumira": 0,
    "Su-E": 0,
    "Son": 0,
    "YC": 0,
}

const logNameCount = (response) => {
    if (response.includes("**Sumira**")) {
        nameCount.Sumira = nameCount.Sumira + 1;
    }
    if (response.includes("\nSumira:")) {
        nameCount.Sumira = nameCount.Sumira - 2;
    }
    if (response.includes("**Su-E**")) {
        nameCount['Su-E'] = nameCount['Su-E'] + 1;
    }
    if (response.includes("\nSu-E:")) {
        nameCount['Su-E'] = nameCount['Su-E'] - 2;
    }
    if (response.includes("**Son**")) {
        nameCount.Sumira = nameCount.Sumira + 1;
    }
    if (response.includes("\nSon:")) {
        nameCount['Son'] = nameCount['Son'] - 2;
    }
    if (response.includes("**YC**")) {
        nameCount.YC = nameCount.YC + 1;
    }
    if (response.includes("\nYC:")) {
        nameCount.YC = nameCount.YC + -2;
    }
    return nameCount;
}

while (true) {
    let this_round_conversation = last_lines ? "Continue" : "Morning";
    numberOfRound = numberOfRound + 1;
    console.log('\x1b[33m', 'Round ', numberOfRound, '\x1b[0m')
    const response = await getResponse(team.David, this_round_conversation);
    logNameCount(response.response);
    this_round_conversation = addToHistory(this_round_conversation, response.response)
    await sleep(2000);
    for (const member of response.target) {
        let res = await getResponse(member, this_round_conversation);
        this_round_conversation = addToHistory(this_round_conversation, res.response)
        logNameCount(res.response);
        await sleep(2000);
    }
    for (const member in nameCount) {
        if (nameCount[member] > 2) {
            if (response.target[response.target.length - 1].name !== member) {
                let res = await getResponse(team[member], this_round_conversation);
                this_round_conversation = addToHistory(this_round_conversation, res.response)
            }
            nameCount[member] = 0
        }
        await sleep(2000);
    }
    const memoryReponse = await memoryChain.call({
        summary: SUMMARY,
        new_lines: this_round_conversation
    })

    SUMMARY = memoryReponse.text;
    last_lines = this_round_conversation;
    await sleep(2000);
}